#include <algorithm>  // std::generate
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <onnxruntime_cxx_api.h>


#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif





/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS( )
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
        return (size_t)0L;      /* Can't open? */
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
    {
        close( fd );
        return (size_t)0L;      /* Can't read? */
    }
    close( fd );
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

using namespace std;

int main(int argc, char** argv) {

  if (argc != 5) {
    cout << "Usage: ./onnx-api-example <onnx_model.onnx> <batch_size> <d_head> <n_head>" << endl;
    return -1;
  }

  char * model_file = argv[1];
  int batch_size = stoi(argv[2]);
  int d_head = stoi(argv[3]);
  int n_head = stoi(argv[4]);


  // onnxruntime setup

    size_t initialMem = getPeakRSS();
  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::Session session(nullptr);
    try {
        session = Ort::Session(env,  model_file, session_options);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }


    int64_t batchSize = batch_size;
    const std::array<int64_t, 2> inputShape = { 1, batchSize };
    const std::array<int64_t, 5> state_shape = { 2, 1, n_head, batchSize, d_head };
    const std::array<int64_t, 2> outputShape = { 1, 10000 };

    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    size_t input_count = session.GetInputCount();
    size_t output_count = session.GetOutputCount();

    for(size_t i=0; i<input_count;i++)
    {
        char * input_name = session.GetInputName(i, alloc);
        inputNames.emplace_back(input_name);
        Ort::TypeInfo info = session.GetInputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> info_shape = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = info_shape.GetShape();
        // cout << "\t" << input_name << " : " << print_shape(input_shape) << endl;
    }

    for(size_t i=0; i<output_count;i++)
    {
        char * input_name = session.GetOutputName(i, alloc);
        outputNames.emplace_back(input_name);
        Ort::TypeInfo info = session.GetInputTypeInfo(i);
        Ort::Unowned<Ort::TensorTypeAndShapeInfo> info_shape = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = info_shape.GetShape();
        // cout << "\t" << input_name << " : " << print_shape(input_shape) << endl;
    }

    std::vector<Ort::Value> inputs;
    std::vector<Ort::Value> outputs;

    inputs.push_back(Ort::Value::CreateTensor<int64_t>(alloc, inputShape.data(), inputShape.size()));
    outputs.push_back(Ort::Value::CreateTensor<float>(alloc, outputShape.data(), outputShape.size()));

    int64_t * input_data = (int64_t *)inputs[0].GetTensorData<int64_t>();
    memset(input_data, 0, batchSize*sizeof(*input_data));

    for (int i=1; i<input_count;i++)
    {
        inputs.push_back(Ort::Value::CreateTensor<float>(alloc, state_shape.data(), state_shape.size()));
        float * data = (float *)inputs.back().GetTensorData<float>();
        memset(data, 0, 2* 1* n_head* batchSize* d_head*sizeof(*data));
        // cout << print_shape(inputs.back().GetTensorTypeAndShapeInfo().GetShape())  << endl;
    }

    Ort::RunOptions runOptions;

    try 
    {
        auto output = session.Run(runOptions, inputNames.data(), inputs.data(), input_count, outputNames.data(), output_count);

        // cout << "After inference: " << getPeakRSS() - initialMem << endl;
        cout << " " << getPeakRSS() << " ";

        // for (size_t i=0; i<output.size();i++)
        // {
        //     cout << print_shape(output[i].GetTensorTypeAndShapeInfo().GetShape())  << endl;
            
        // }
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

}
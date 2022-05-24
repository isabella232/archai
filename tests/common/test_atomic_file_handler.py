# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from logging import LogRecord

from archai.common.atomic_file_handler import AtomicFileHandler

FILENAME = "file"
ENCODING = None
SAVE_DELAY = 30.0


def test_atomic_file_handler_class_init():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )

    assert isinstance(atomic_file_handler._buffer, list)
    assert atomic_file_handler._last_flush == 0.0

    assert os.path.basename(atomic_file_handler.baseFilename) == FILENAME
    assert atomic_file_handler.encoding is ENCODING
    assert atomic_file_handler.save_delay == SAVE_DELAY


def test_atomic_file_handler_class_flush():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )

    atomic_file_handler.flush()


def test_atomic_file_handler_class_close():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )

    atomic_file_handler.close()


def test_atomic_file_handler_class_open():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )

    atomic_file_handler._open()


def test_atomic_file_handler_class_flush_buffer():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )

    atomic_file_handler._flush_buffer(force=False)
    atomic_file_handler._flush_buffer(force=True)


def test_atomic_file_handler_class_emit():
    atomic_file_handler = AtomicFileHandler(
        FILENAME, encoding=ENCODING, save_delay=SAVE_DELAY
    )
    log_record = LogRecord("name", 0, "pathname", 0, "msg", None, None)

    atomic_file_handler.emit(log_record)

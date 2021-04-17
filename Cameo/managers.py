# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import time


class CaptureManager(object):

    def __init__(self, capture, preview_window_manager=None,
                 should_mirror_preview=False):
        self.preview_window_manager = preview_window_manager
        self.should_mirror_preview = should_mirror_preview

        self._capture = capture
        self._channel = 0
        self._entered_frame = False
        self._frame = None
        self._image_filename = None
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

        self._start_time = None
        self._frames_elapsed = np.long(0)
        self._fps_estimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._entered_frame and self._frame is None:
            _, self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        return self._image_filename is not None

    @property
    def isWritingVideo(self):
        return self._video_filename is not None

    def enterFrame(self):
        """Capture the next frame, if any."""

        # But first, check that any previous frame was exited.
        assert not self._entered_frame, 'previous enterFrame() had no ' \
                                        'matching exitFrame() '

        if self._capture is not None:
            self._entered_frame = self._capture.grad()

    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._entered_frame = False
            return

        # Update the FPS estimate and related variables
        if self._frames_elapsed == 0:
            self._start_time = time.time()
        else:
            time_elapsed = time.time() - self._start_time
            self._fps_estimate = self._frames_elapsed / time_elapsed  # trouble
            self._frames_elapsed += 1  # trouble

        # Draw to the window, if any.
        if self.preview_window_manager is not None:
            if self.should_mirror_preview:
                mirrored_frame = np.fliplr(self._frame).copy()
                self.preview_window_manager.show(mirrored_frame)
            else:
                self.preview_window_manager.show(self._frame)

        # Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._image_filename, self._frame)
            self._image_filename = None

        # Write to the video file, if any.
        self._writeVideoFrame()

        # Release the frame
        self._frame = None
        self._entered_frame = False

    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._image_filename = filename

    def startWritingVideo(self, filename,
                          encoding=cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        """Start writing exited frames to a video file."""
        self._video_filename = filename
        self._video_encoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frames to a video file."""
        self._video_filename = None
        self._video_encoding = None
        self._video_writer = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return

        if self._video_writer is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                # The capture's FPS is unknown so use an estimate
                if self._frames_elapsed < 2.0:
                    # Wait until more frames elapse so that the estimate is
                    # more stable.
                    return
                else:
                    fps = self._fps_estimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._video_writer = cv2.VideoWriter(self._video_filename,
                                                 self._video_encoding,
                                                 fps, size)
        self._video_writer.write(self._frame)

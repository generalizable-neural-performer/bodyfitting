import struct

import cv2
import numpy as np


class StreamerFileDecoder(object):

    def __init__(self, rgbd_path, debug_flag=False):
        super().__init__()
        self.debug = debug_flag
        self.rgbd_path = rgbd_path
        self.file_stream = open(self.rgbd_path, 'rb')
        self.__size_of_int__ = 4  # bytes
        self.__size_of_uint16__ = 2
        self.__size_of_float__ = 4  # bytes
        self.__size_of_size_t__ = 8  # bytes
        self.__fileParse__()

    def getFrame(self, frame_index):
        if frame_index < len(self.frame_offset_in_bytes):
            frame_offset = self.frame_offset_in_bytes[frame_index]
            self.file_stream.seek(frame_offset)
            reader_color_timestamp = self.__getOneSize_t__()
            color_buffer_size = self.__getOneSize_t__()
            color_buffer = self.__getRawBytes__(color_buffer_size)
            encoded_image = np.asarray(bytearray(color_buffer), dtype='uint8')
            color_image_mat = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
            reader_depth_timestamp = self.__getOneSize_t__()
            depth_buffer = self.__getRawBytes__(
                self.depth_size)  # variable type: cv::Mat::data in Cpp
            index_buffer = self.__getRawBytes__(
                self.index_size)  # varaiable type: cv::Mat::data in Cpp

            # src_depth_image_mat = np.zeros(
            #     (self.depth_height, self.depth_width, self.depth_channels),
            #     np.uint8)
            # src_depth_image_mat.data = bytearray(depth_buffer)
            # #src_depth_image_mat = src_depth_image_mat #/ 20.0
            # depth_8uc2 = np.uint8(src_depth_image_mat)

            depth_8uc2 = np.asarray(
                bytearray(depth_buffer), dtype='uint8').reshape(
                    (self.depth_height, self.depth_width, self.depth_channels))
            index_8uc1 = np.asarray(
                bytearray(index_buffer), dtype='uint8').reshape(
                    (self.depth_height, self.depth_width))
            depth_16uc1 = np.zeros((depth_8uc2.shape[0], depth_8uc2.shape[1]),
                                   np.uint16)
            depth_16uc1.data = bytearray(depth_buffer)

            if self.debug:
                print('color_buffer_size:', color_buffer_size)
                print('self.depth_size:', self.depth_size)
                print('self.index_size:', self.index_size)
            return [
                color_image_mat, reader_color_timestamp, depth_8uc2,
                depth_16uc1, reader_depth_timestamp, index_8uc1
            ]
        else:
            return None

    def __fileParse__(self):
        self.frame_count = self.__getOneInt__()
        if self.frame_count == 0:
            self.frame_count = 9999
        self.color_width = self.__getOneInt__()
        self.color_height = self.__getOneInt__()
        self.color_channels = self.__getOneInt__()
        if self.debug:
            print('self.color_width: ', self.color_width)
            print('self.color_height: ', self.color_height)
            print('self.color_channels: ', self.color_channels)

        self.depth_width = self.__getOneInt__()
        self.depth_height = self.__getOneInt__()
        self.depth_channels = self.__getOneInt__()
        if self.debug:
            print('self.depth_width: ', self.depth_width)
            print('self.depth_height: ', self.depth_height)
            print('self.depth_channels: ', self.depth_channels)
        self.depth_size = (
            self.depth_width * self.depth_height * self.depth_channels)
        self.index_size = self.depth_width * self.depth_height

        self.color_camera_intrinsics_dict = self.__intrinsicFread__()
        self.depth_camera_intrinsics_dict = self.__intrinsicFread__()
        self.extrinsics_dict = self.__extrinsicFread__()

        headerBytesSize = 7 * self.__size_of_int__ + 2 * (
            9 + 10) * self.__size_of_float__ + 16 * self.__size_of_float__
        self.frame_offset_in_bytes = []
        self.frame_offset_in_bytes.append(headerBytesSize)
        for i in range(1, self.frame_count, 1):
            previous_offset = self.frame_offset_in_bytes[-1]
            self.file_stream.seek(previous_offset, 0)
            color_ts = self.__getOneSize_t__()
            color_sz = self.__getOneSize_t__()
            previous_frame_bytes = (3 * self.__size_of_size_t__ + color_sz +
                                    self.depth_size + self.index_size)
            if self.debug:
                print("previous_frame_bytes: ", previous_frame_bytes)
            current_frame_offset = previous_offset + previous_frame_bytes
            self.frame_offset_in_bytes.append(current_frame_offset)

    def __intrinsicFread__(self):
        ret_dict = {}
        intrinsic_mat = np.zeros(shape=[3, 3], dtype=np.float32)
        for i in range(intrinsic_mat.shape[0]):
            for j in range(intrinsic_mat.shape[1]):
                intrinsic_mat[i][j] = self.__getOneFloat__()
        ret_dict['in_mat'] = intrinsic_mat
        ret_dict['k1'] = self.__getOneFloat__()
        ret_dict['k2'] = self.__getOneFloat__()
        ret_dict['k3'] = self.__getOneFloat__()
        ret_dict['k4'] = self.__getOneFloat__()
        ret_dict['k5'] = self.__getOneFloat__()
        ret_dict['k6'] = self.__getOneFloat__()
        ret_dict['p1'] = self.__getOneFloat__()
        ret_dict['p2'] = self.__getOneFloat__()
        ret_dict['codx'] = self.__getOneFloat__()
        ret_dict['cody'] = self.__getOneFloat__()
        return ret_dict

    def __extrinsicFread__(self):
        ret_dict = {}
        extrinsic_mat = np.zeros(shape=[4, 4], dtype=np.float32)
        for i in range(extrinsic_mat.shape[0]):
            for j in range(extrinsic_mat.shape[1]):
                extrinsic_mat[i][j] = self.__getOneFloat__()
        ret_dict['depth2color_mat'] = extrinsic_mat
        ret_dict['depth2color_rotation'] = np.zeros(
            shape=[
                9,
            ], dtype=np.float32)
        ret_dict['depth2color_translation'] = np.zeros(
            shape=[
                3,
            ], dtype=np.float32)
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                ret_dict['depth2color_rotation'][i * 3 + j] = extrinsic_mat[i,
                                                                            j]
            ret_dict['depth2color_translation'][i] = extrinsic_mat[i, 3]
        return ret_dict

    def __getOneInt__(self):
        ret_int = struct.unpack('i',
                                self.file_stream.read(self.__size_of_int__))[0]
        return ret_int

    def __getOneFloat__(self):
        ret_float = struct.unpack(
            'f', self.file_stream.read(self.__size_of_float__))[0]
        return ret_float

    def __getOneSize_t__(self):
        # self.file_stream.read(self.__size_of_size_t__)
        ret_size_t = struct.unpack(
            'N', self.file_stream.read(self.__size_of_size_t__))[0]
        return ret_size_t

    def __getRawBytes__(self, byte_size=1):
        return self.file_stream.read(byte_size)

    def writeDepthFile(self, depth_file_path=None):
        if depth_file_path is None:  # .rgbd->.depth
            depth_file_path = self.rgbd_path[:-4] + 'depth'
        if not depth_file_path.endswith('.depth'):
            depth_file_path += '.depth'
        with open(depth_file_path, 'wb') as f_write:
            self.file_stream.seek(0)
            head_size = self.frame_offset_in_bytes[0]
            head_buffer = self.__getRawBytes__(head_size)
            f_write.write(head_buffer)
            for frame_index in range(len(self.frame_offset_in_bytes)):
                frame_offset = self.frame_offset_in_bytes[frame_index]
                frame_depth_data = self.__assembleRawDepthData__(frame_offset)
                f_write.write(frame_depth_data)

    def __assembleRawDepthData__(self, frame_offset):
        self.file_stream.seek(frame_offset)
        # __reader_color_timestamp = self.__getOneSize_t__()
        # __color_buffer_size = self.__getOneSize_t__()
        # __color_buffer = self.__getRawBytes__(__color_buffer_size)
        # read depth head and data in one array
        all_depth_buffer = self.__getRawBytes__(self.__size_of_size_t__ +
                                                self.depth_size)
        # # read all data(including color head and mat), only for test
        # self.file_stream.seek(frame_offset)
        # all_depth_buffer = self.__getRawBytes__(3*self.__size_of_size_t__ +
        #           __color_buffer_size + self.depth_size + self.index_size)
        return all_depth_buffer

    def close(self):
        self.file_stream.close()

# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from opencv_apps/Moment.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import opencv_apps.msg

class Moment(genpy.Message):
  _md5sum = "560ee3fabfffb4ed4155742d6db8a03c"
  _type = "opencv_apps/Moment"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """# spatial moments
float64 m00
float64 m10
float64 m01
float64 m20
float64 m11
float64 m02
float64 m30
float64 m21
float64 m12
float64 m03

# central moments
float64 mu20
float64 mu11
float64 mu02
float64 mu30
float64 mu21
float64 mu12
float64 mu03

# central normalized moments
float64 nu20
float64 nu11
float64 nu02
float64 nu30
float64 nu21
float64 nu12
float64 nu03

# center of mass m10/m00, m01/m00
Point2D center
float64 length
float64 area

================================================================================
MSG: opencv_apps/Point2D
float64 x
float64 y

"""
  __slots__ = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03','mu20','mu11','mu02','mu30','mu21','mu12','mu03','nu20','nu11','nu02','nu30','nu21','nu12','nu03','center','length','area']
  _slot_types = ['float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','float64','opencv_apps/Point2D','float64','float64']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       m00,m10,m01,m20,m11,m02,m30,m21,m12,m03,mu20,mu11,mu02,mu30,mu21,mu12,mu03,nu20,nu11,nu02,nu30,nu21,nu12,nu03,center,length,area

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Moment, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.m00 is None:
        self.m00 = 0.
      if self.m10 is None:
        self.m10 = 0.
      if self.m01 is None:
        self.m01 = 0.
      if self.m20 is None:
        self.m20 = 0.
      if self.m11 is None:
        self.m11 = 0.
      if self.m02 is None:
        self.m02 = 0.
      if self.m30 is None:
        self.m30 = 0.
      if self.m21 is None:
        self.m21 = 0.
      if self.m12 is None:
        self.m12 = 0.
      if self.m03 is None:
        self.m03 = 0.
      if self.mu20 is None:
        self.mu20 = 0.
      if self.mu11 is None:
        self.mu11 = 0.
      if self.mu02 is None:
        self.mu02 = 0.
      if self.mu30 is None:
        self.mu30 = 0.
      if self.mu21 is None:
        self.mu21 = 0.
      if self.mu12 is None:
        self.mu12 = 0.
      if self.mu03 is None:
        self.mu03 = 0.
      if self.nu20 is None:
        self.nu20 = 0.
      if self.nu11 is None:
        self.nu11 = 0.
      if self.nu02 is None:
        self.nu02 = 0.
      if self.nu30 is None:
        self.nu30 = 0.
      if self.nu21 is None:
        self.nu21 = 0.
      if self.nu12 is None:
        self.nu12 = 0.
      if self.nu03 is None:
        self.nu03 = 0.
      if self.center is None:
        self.center = opencv_apps.msg.Point2D()
      if self.length is None:
        self.length = 0.
      if self.area is None:
        self.area = 0.
    else:
      self.m00 = 0.
      self.m10 = 0.
      self.m01 = 0.
      self.m20 = 0.
      self.m11 = 0.
      self.m02 = 0.
      self.m30 = 0.
      self.m21 = 0.
      self.m12 = 0.
      self.m03 = 0.
      self.mu20 = 0.
      self.mu11 = 0.
      self.mu02 = 0.
      self.mu30 = 0.
      self.mu21 = 0.
      self.mu12 = 0.
      self.mu03 = 0.
      self.nu20 = 0.
      self.nu11 = 0.
      self.nu02 = 0.
      self.nu30 = 0.
      self.nu21 = 0.
      self.nu12 = 0.
      self.nu03 = 0.
      self.center = opencv_apps.msg.Point2D()
      self.length = 0.
      self.area = 0.

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_28d().pack(_x.m00, _x.m10, _x.m01, _x.m20, _x.m11, _x.m02, _x.m30, _x.m21, _x.m12, _x.m03, _x.mu20, _x.mu11, _x.mu02, _x.mu30, _x.mu21, _x.mu12, _x.mu03, _x.nu20, _x.nu11, _x.nu02, _x.nu30, _x.nu21, _x.nu12, _x.nu03, _x.center.x, _x.center.y, _x.length, _x.area))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      if self.center is None:
        self.center = opencv_apps.msg.Point2D()
      end = 0
      _x = self
      start = end
      end += 224
      (_x.m00, _x.m10, _x.m01, _x.m20, _x.m11, _x.m02, _x.m30, _x.m21, _x.m12, _x.m03, _x.mu20, _x.mu11, _x.mu02, _x.mu30, _x.mu21, _x.mu12, _x.mu03, _x.nu20, _x.nu11, _x.nu02, _x.nu30, _x.nu21, _x.nu12, _x.nu03, _x.center.x, _x.center.y, _x.length, _x.area,) = _get_struct_28d().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_28d().pack(_x.m00, _x.m10, _x.m01, _x.m20, _x.m11, _x.m02, _x.m30, _x.m21, _x.m12, _x.m03, _x.mu20, _x.mu11, _x.mu02, _x.mu30, _x.mu21, _x.mu12, _x.mu03, _x.nu20, _x.nu11, _x.nu02, _x.nu30, _x.nu21, _x.nu12, _x.nu03, _x.center.x, _x.center.y, _x.length, _x.area))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      if self.center is None:
        self.center = opencv_apps.msg.Point2D()
      end = 0
      _x = self
      start = end
      end += 224
      (_x.m00, _x.m10, _x.m01, _x.m20, _x.m11, _x.m02, _x.m30, _x.m21, _x.m12, _x.m03, _x.mu20, _x.mu11, _x.mu02, _x.mu30, _x.mu21, _x.mu12, _x.mu03, _x.nu20, _x.nu11, _x.nu02, _x.nu30, _x.nu21, _x.nu12, _x.nu03, _x.center.x, _x.center.y, _x.length, _x.area,) = _get_struct_28d().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_28d = None
def _get_struct_28d():
    global _struct_28d
    if _struct_28d is None:
        _struct_28d = struct.Struct("<28d")
    return _struct_28d
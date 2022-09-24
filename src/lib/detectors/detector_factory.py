from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .perception import MultiPoseDetector

detector_factory = {
  'perception': MultiPoseDetector, 
}

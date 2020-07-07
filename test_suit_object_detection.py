import unittest
from test_objectdetection import Object_Detection_TestCase

def suite():

   suite = unittest.TestSuite()
   suite.addTests( unittest.TestLoader().loadTestsFromTestCase(Object_Detection_TestCase))
   return suite

if __name__ == '__main__':
   unittest.TextTestRunner(verbosity=2).run(suite())

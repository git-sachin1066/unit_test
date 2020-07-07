from test_classifier import Classifier
import unittest



def suite():

   suite = unittest.TestSuite()
   suite.addTests( unittest.TestLoader().loadTestsFromTestCase(Classifier))
   return suite

if __name__ == '__main__':
   unittest.TextTestRunner(verbosity=2).run(suite())
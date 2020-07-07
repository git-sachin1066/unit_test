import unittest, os , cv2
from shutil import rmtree
from mock import patch
from test import main,cli,convertPDFToPNG
class Object_Detection_TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_path = [os.path.join(os.path.dirname(__file__),'test_pdf_output','test_pdf.pdf')]
        cls.output = os.path.join(os.path.dirname(__file__),'test_pdf_output','output')
        if os.path.exists(cls.output) and os.path.exists(cls.pdf_path[0]):
            pass
        else :
            print('test items not found')
            exit(0)


    '''call main with actual data'''


    def test_convertPDFToPNG(self):
         self.ret_path = convertPDFToPNG(self.pdf_path[0],self.output)
         self.assertTrue(os.path.exists(self.ret_path[0]))
         self.assertEqual(os.path.splitext(self.ret_path[0])[1], '.png')
         self.img = cv2.imread(self.ret_path[0])
         self.assertIsNotNone((self.img))
         os.remove(self.ret_path[0])
    def test_predict(self):
        self.json_lumi = cli.predict( self.output)
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(self.output),'result.json')))
        self.assertIsInstance(self.json_lumi,list)
        self.assertEqual(list(self.json_lumi[0].keys()),['file','objects','page_no'])
        os.remove(os.path.join(os.path.dirname(self.output),'result.json'))
    @classmethod
    def tearDownClass(cls):
        del cls.output,cls.pdf_path

if __name__ == '__main__':
    unittest.main()

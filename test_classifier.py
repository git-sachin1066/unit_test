import unittest
import predict , numpy
from models.loader import get_model
import os ,  tensorflow
from unittest.mock import MagicMock
from mock import patch
import mock



class Classifier(unittest.TestCase):
    """setupClass will do the need ful at the very fosrst step
    of the program and th eoutput are aviliable until the code executes"""
    @classmethod
    def setUpClass(cls):
        cls.path_model = get_model()
        #cls.image_folder = os.path.join(os.path.dirname(__file__),r'ocr/table_ocr/test_images_json/table.png')
        cls.image_path_list = [os.path.join(os.path.dirname(os.path.dirname(__file__)),r'ocr\table_ocr\test_images_json\table1_table.png')]

    ## check the datatype returned from predict_with_model
    @patch('tensorflow.keras.Model.predict')
    def test_method_predict1(self, mock_get_modelpredict):
        prob_array = numpy.random.uniform(low=0, high=1, size=(1, 5)).astype(float)
        mock_get_modelpredict.return_value = prob_array
        self.assertIs(type(predict.predict_with_model(self.image_path_list, self.path_model)), list)
    ## manually check for a given label of invoice type
    @patch('tensorflow.keras.Model.predict')
    def test_method_predict2(self,mock_get_modelpredict):
        ## checking for invoice type ##
        ##ground truth ---> trained_label = ["invoice", "ic", "guarantee-letter", "authorization-form", "lab-report"]
        prob_array = numpy.array([[1, 0,0,0,0,]])
        #mock_get_model.return_value = tensorflow.python.keras.engine.sequential.Sequential
        mock_get_modelpredict.return_value = prob_array
        result = predict.predict_with_model(self.image_path_list, self.path_model)
        self.assertEqual(result[0]['label'] , 'invoice')

    ## check for the max probability is > 0.85 or not
    @patch('tensorflow.keras.Model.predict')
    def test_method_predict4(self, mock_get_modelpredict):
        ## checking for invoice type ##
        ##ground truth ---> trained_label = ["invoice", "ic", "guarantee-letter", "authorization-form", "lab-report"]
        prob_array = numpy.array([[1, 0, 0, 0, 0, ]])
        # mock_get_model.return_value = tensorflow.python.keras.engine.sequential.Sequential
        mock_get_modelpredict.return_value = prob_array
        result = predict.predict_with_model(self.image_path_list, self.path_model)
        self.assertGreaterEqual(result[0]['probability'],0.85)

    ## check for the image folder contains only .png or not
    def test_method_predict3(self):##test scenario
        for item in self.image_path_list[:] :
            self.assertIn('.png', item )


    ## check for the values,
    ## if input image path list is empty
    ## if input path list is string  ---> it led to if else in predict_with_model
    def test_values(self):
        self.assertRaises(ValueError,predict.predict_with_model,[],self.path_model) ## check for empty list
        self.assertRaises(ValueError, predict.predict_with_model, str(''),self.path_model) ## check for type of the argument

    ###tearDownClass will be the last step odf the test case
    @classmethod
    def tearDownClass(cls):
        del cls.path_model, cls.image_path_list

if __name__ == '__main__':
    unittest.main()




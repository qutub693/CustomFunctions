import logging
import numpy as np
import pandas as pd
import scipy as sp
import logging
import iotfunctions
from iotfunctions.base import (BaseTransformer, BaseRegressor, BaseEstimatorFunction, BaseSimpleAggregator)
from iotfunctions.bif import (AlertHighValue)
from iotfunctions.ui import (UISingle, UIMulti, UIMultiItem, UIFunctionOutSingle, UISingleItem, UIFunctionOutMulti)
from iotfunctions.dbtables import (FileModelStore, DBModelStore)

from .anomaly import SupervisedLearningTransformer

logger = logging.getLogger(__name__)
logger.info('IOT functions version ' + iotfunctions.__version__)

PACKAGE_URL = 'git+https://github.com/qutub693/CustomFunctions.git'
_IS_PREINSTALLED = False


class VerySimpleModel:
    def __init__(self):
        self.Anomaly=0
    

class SuddenSpikesV2(SupervisedLearningTransformer):

    def __init__(self, input_item,Anomaly):
        super().__init__(features=[input_item], targets=[Anomaly])

        self.input_item = input_item
        self.Anomaly = Anomaly
        self.auto_train = True
        self.whoami = 'SuddenSpikesV2'

    def execute(self, df):
        # set output columns to zero
        logger.debug('Called ' + self.whoami + ' with columns: ' + str(df.columns))
        df[self. Anomaly] = 0
        return super().execute(df)

    def _calc(self, df):
        entity = df.index[0][0]

        # obtain db handler
        db = self.get_db()
        test_model_name=self.get_model_name(suffix=entity)

#         log_stuff = 'Name of the model:' + str(test_model_name) + ', Entity Value: ' + str(entity) + ', Entity Type ' + str(self.get_entity_type())
#         logger.info(log_stuff)
#         raise Exception(log_stuff)
        model_name, very_simple_model, version = self.load_model(suffix=entity)
        feature = df[self.input_item].values

        if very_simple_model is None and self.auto_train:
            print('Here 1')

            # we don't do that now, the model *has* to be there
            very_simple_model = VerySimpleModel()

            try:
                db.model_store.store_model(model_name, very_simple_model)
            except Exception as e:
                logger.error('Model store failed with ' + str(e))

            print('Here')
        else:
            print('Here 5')
        print(very_simple_model)


        if very_simple_model is not None:
            #self.Min[entity] = very_simple_model.Min
            #df[self.Anomaly] = very_simple_model.Anomaly      # set the anomaly column
            #self.Max[entity] = very_simple_model.Max
            #df[self.Max] = very_simple_model.Max       # set the max threshold column
            #df[self.outlier] = np.logical_and(feature < very_simple_model.Max, feature > very_simple_model.Min)
            df[self.Anomaly] = np.where(feature >6000,feature,0)

        return df.droplevel(0)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=float, description="Data item to analyze"))
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="Anomaly", datatype=float,
                                           description="Boolean outlier condition"))

        return (inputs, outputs)
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
    def __init__(self, total):
        self.total = total

class AlertRollUpFunctionV2(SupervisedLearningTransformer):

    def __init__(self, total):
        super().__init__(features=[input_item], targets=[total])

        self.input_item = input_item
        self.total = total
        self.auto_train = True
        self.whoami = 'AlertRollUpFunctionV2'


    def execute(self, df):
        # set output columns to zero
        logger.debug('Called ' + self.whoami + ' with columns: ' + str(df.columns))
        df[self.total] = 0
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

        # feature = df[self.input_item].values

        if very_simple_model is None and self.auto_train:
            print('Here 1')

            # we don't do that now, the model *has* to be there
            very_simple_model = VerySimpleModel(91943)

            try:
                db.model_store.store_model(model_name, very_simple_model)
            except Exception as e:
                logger.error('Model store failed with ' + str(e))
        else:
            print('Here 5')
        print(very_simple_model)

        if very_simple_model is not None:
            df[self.total] = very_simple_model.total 

        return df.droplevel(0)

    @classmethod
    def build_ui(cls):
        # define arguments that behave as function inputs
        inputs = []
        inputs.append(UISingleItem(name="input_item", datatype=None, description="Data item to analyze"))
        # define arguments that behave as function outputs
        outputs = []
        outputs.append(UIFunctionOutSingle(name="total", datatype=None,
                                           description="Total Alerts"))
        return (inputs, outputs)
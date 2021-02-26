import {combineReducers} from 'redux';

import annotations, { spanAnnotations,highlights, selectedLink, ners } from './annotationsReducer'
import concepts, { selectedConcept } from './conceptsReducer'
import connectivesKeyTypes from           './ConnectivesKeyTypesReducer'
import datasets from                      './datasetsReducer'
import { label, suggestedLF } from        './labelAndSuggestLFReducer'
import labelClasses from                  './labelClassesReducer'
import labelExamples from                 './labelExampleReducer'
import loadingBarReducer from             './loadingBarReducer'
import selectedLF from                    './selectedLFReducer'
import statistics, {statistics_LRmodel} from './statisticsReducer'
import text from                          './textReducer'

const rootReducer = combineReducers({
  annotations,
  spanAnnotations,
  concepts,
  datasets,
  gll: connectivesKeyTypes,
  highlights,
  label,
  labelClasses,
  labelExamples,
  launchProgress: loadingBarReducer,
  ners,
  selectedConcept,
  selectedLF,
  selectedLink,
  suggestedLF,
  statistics,
  statistics_LRmodel,
  text
});

export default rootReducer;


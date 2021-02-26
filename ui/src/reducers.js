import {combineReducers} from 'redux';

import annotations, { selectedLink } from './annotationsReducer'
import concepts, { selectedConcept } from './conceptsReducer'
import connectivesKeyTypes from           './ConnectivesKeyTypesReducer'
import interactionHistory from            './interactionHistoryReducer'
import { label, suggestedLF } from        './labelAndSuggestLFReducer'
import labelClasses from                  './labelClassesReducer'
import labelExamples from                 './labelExampleReducer'
import selectedLF from                    './selectedLFReducer'
import statistics from                    './statisticsReducer'
import text from                          './textReducer'

const rootReducer = combineReducers({
  annotations,
  concepts,
  gll: connectivesKeyTypes,
  interactionHistory,
  label,
  labelClasses,
  labelExamples,
  selectedConcept,
  selectedLF,
  selectedLink,
  suggestedLF,
  statistics,
  text
});

export default rootReducer;


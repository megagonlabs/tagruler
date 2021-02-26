
import { 
    GET_CONCEPTS_PENDING, 
    UPDATE_CONCEPT_PENDING, 
    GET_CONCEPTS_SUCCESS, 
    GET_CONCEPTS_ERROR,
    SELECT_CONCEPT
} from '../actions/concepts'

const initialState = {
    pending: false,
    data: {},
    error: null
}

function  concepts (state = initialState, action ){
    switch (action.type) {
        case GET_CONCEPTS_PENDING:
            return {
                ...state,
                data: {...state.data, ...action.data},
                pending: true
            }
        case UPDATE_CONCEPT_PENDING:
            const conceptName = action.conceptName;
            state.data[conceptName].pending = true;
            return {
                ...state,
                data: state.data,
                pending: false
            }
        case GET_CONCEPTS_SUCCESS:
            return {
                ...state,
                data: action.data,
                pending: false,
            }  
        case GET_CONCEPTS_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default:
            return state
     }
 }

export default concepts;

export function selectedConcept(state=null, action) {
    switch (action.type) {
      case SELECT_CONCEPT: 
        return action.data
        //{...state, ...action.data}
      default: 
        return state
    }
 }

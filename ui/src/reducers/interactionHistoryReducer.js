
import { 
    GET_INTERACTION_SUCCESS, 
    GET_INTERACTION_PENDING, 
    GET_INTERACTION_ERROR 
} from '../actions/interaction'

const initialState = {
    pending: false,
    data: {},
    error: null
}

function interactionHistory(state = initialState, action ){
    switch (action.type) {
        case GET_INTERACTION_PENDING:
            return {
                ...state,
                data: {},
                pending: true
            }
        case GET_INTERACTION_SUCCESS:
            return {
                ...state,
                data: action.data,
                pending: false,
            }  
        case GET_INTERACTION_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default:
            return state
     }
 }

 export default interactionHistory;
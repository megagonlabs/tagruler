import { 
    GET_CLASSES_SUCCESS,
    GET_CLASSES_PENDING,
    GET_CLASSES_ERROR,
    ADD_CLASS_SUCCESS
} from '../actions/labelClasses'

const initialState = {
    pending: true,
    data: [],
    error: null
}

function labelClassesReducer(state=initialState, action){
    switch (action.type) {
        case GET_CLASSES_SUCCESS:
            return {
                ...state,
                data: action.data,
                pending: false
            }
        case GET_CLASSES_ERROR:
            return {
                ...state,
                error: action.error,
                pending: false
            }
        case GET_CLASSES_PENDING:
            return {
                ...state,
                pending: true
            }
        case ADD_CLASS_SUCCESS:    
            return {
                ...state,
                data: [...state.data, action.data],
                pending: false
            }        
        default:
            return state;
    }
}

export default labelClassesReducer;
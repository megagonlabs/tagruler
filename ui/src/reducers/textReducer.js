import { 
    GET_TEXT_PENDING, 
    GET_TEXT_SUCCESS, 
    GET_TEXT_ERROR
} from '../actions/getText';

const initialState = {
    pending: false,
    data: "",
    error: null
}

function text (state =initialState, action ){
    switch (action.type) {
        case GET_TEXT_PENDING: 
            return {
                ...state,
                pending: true
            }   
        case GET_TEXT_SUCCESS:
        return {
            ...state,
            data: action.data,
            index: action.index,
            pending: false
        } 
        case GET_TEXT_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default: 
            return state;
    }
}

export default text;
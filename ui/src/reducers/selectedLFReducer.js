import { 
  SUBMIT_LF_PENDING, 
  SUBMIT_LF_SUCCESS, 
  SUBMIT_LF_ERROR, 
  ONE_LF_PENDING,
  LF_STATS, 
  LF_STATS_ERROR 
} from '../actions/submitLFs'

const initialState = {
    pending: false,
    data: {},
    error: null
}

function updateData(old_data, new_data) {
    old_data = {...old_data, ...new_data}
    return Object.keys(old_data).reduce(function(updated_data, key) {
        updated_data[key] = old_data[key];
        return updated_data;
    }, {});
}

export default function selectedLFReducer(state=initialState, action) {
    switch (action.type) {
        case SUBMIT_LF_PENDING:
            return {
                ...state,
                data: {...state.data, ...action.data},
                pending: true
            }
        case ONE_LF_PENDING:
            return state // TODO tell redux it's loading
        case SUBMIT_LF_SUCCESS:
            return {
                ...state,
                data: updateData(state.data, action.data),
                pending: false,
                stats: false
            }
        case LF_STATS:
            return {
                data: action.data,
                pending: false,
                stats: true
            }
        case LF_STATS_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        case SUBMIT_LF_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default:
            return state;
    }
}
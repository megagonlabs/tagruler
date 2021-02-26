import { GET_STATS_PENDING, GET_STATS_SUCCESS, GET_STATS_ERROR } from '../actions/getStatistics'
import { GET_LRSTATS_PENDING, GET_LRSTATS_SUCCESS, GET_LRSTATS_ERROR } from '../actions/getStatistics'


const initialState = {
    pending: false,
    data: {},
    error: null
}

export default function statisticsReducer(state=initialState, action) {
    switch (action.type) {
        case GET_STATS_PENDING:
            return {
                ...state,
                pending: true
            }
        case GET_STATS_SUCCESS:
            return {
                ...state,
                data: action.data,
                pending: false
            }
        case GET_STATS_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default:
            return state;
    }
}


export function statistics_LRmodel(state=initialState, action) {
    switch (action.type) {
        case GET_LRSTATS_PENDING:
            return {
                ...state,
                pending: true
            }
        case GET_LRSTATS_SUCCESS:
            return {
                ...state,
                data: action.data,
                pending: false
            }
        case GET_LRSTATS_ERROR:
            return {
                ...state,
                pending: false,
                error: action.error
            }
        default:
            return state;
    }
}
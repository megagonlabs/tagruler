import axios from 'axios';
import api from './api'

export const GET_STATS_PENDING = "GET_STATS_PENDING";
export const GET_STATS_SUCCESS = "GET_STATS_SUCCESS";
export const GET_STATS_ERROR = "GET_STATS_ERROR";

export function getStatsPending() {
    return {
        type: GET_STATS_PENDING
    }
}

function getStatsSuccess(data) {
    return {
        type: GET_STATS_SUCCESS,
        data: data
    }
}

function getStatsError(error) {
    return {
        type: GET_STATS_ERROR,
        error: error
    }
}

function getStatistics() {
    return dispatch => {
        dispatch(getStatsPending());
        axios.get(`${api}/statistics`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            dispatch(getStatsSuccess(response.data));
            return response.data;
        })
        .catch(error => {
            dispatch(getStatsError(error));
        })
    }
}

export default getStatistics;


export const GET_LRSTATS_PENDING = "GET_LRSTATS_PENDING";
export const GET_LRSTATS_SUCCESS = "GET_LRSTATS_SUCCESS";
export const GET_LRSTATS_ERROR = "GET_LRSTATS_ERROR";

function getLRStatsPending() {
    return {
        type: GET_LRSTATS_PENDING
    }
}

function getLRStatsSuccess(data) {
    return {
        type: GET_LRSTATS_SUCCESS,
        data: data
    }
}

function getLRStatsError(error) {
    return {
        type: GET_LRSTATS_ERROR,
        error: error
    }
}

export function getLRStatistics() {
    return dispatch => {
        dispatch(getLRStatsPending());
        axios.get(`${api}/lr_statistics`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            dispatch(getLRStatsSuccess(response.data));
            return response.data;
        })
        .catch(error => {
            dispatch(getLRStatsError(error));
        })
    }
}
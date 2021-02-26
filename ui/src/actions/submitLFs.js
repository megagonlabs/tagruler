import axios from 'axios';
import api from './api'
import getStatistics from './getStatistics'


export const SUBMIT_LF_PENDING = "SUBMIT_LF_PENDING";
export const ONE_LF_PENDING = "ONE_LF_PENDING";
export const SUBMIT_LF_SUCCESS = "SUBMIT_LF_SUCCESS";
export const SUBMIT_LF_ERROR = "SUBMIT_LF_ERROR";
export const LF_STATS = "LF_STATS";
export const LF_STATS_ERROR = "LF_STATS_ERROR";
export const LF_LABEL_EXAMPLES = "LF_LABEL_EXAMPLES";


function allLFPending(data) {
    if (data) {
        return {
            type: SUBMIT_LF_PENDING,
            data: data
        }
    } return { type: SUBMIT_LF_PENDING }

}

function oneLFPending(lf_id) {
    return {
        type: ONE_LF_PENDING,
        lf_id: lf_id
    }
}

function submitLFSuccess(data) {
    return {
        type: SUBMIT_LF_SUCCESS,
        data: data
    }
}

function submitLFError(error) {
    return {
        type: SUBMIT_LF_ERROR,
        error: error
    }
}

function lfStats(data) {
    return {
        type: LF_STATS,
        data: data
    }
}

function lfStatsError(error) {
    return {
        type: LF_STATS_ERROR,
        error: error
    }
}

export function deleteLF(lf_ids) {
    return dispatch => {
        dispatch(allLFPending());
        axios.put(`${api}/labelingfunctions`, lf_ids)
        .then(response => {
            dispatch(getLFstats());
        })
    }
}

function submitLFs(data) {
    return dispatch => {
        dispatch(allLFPending(data));
        axios.put(`${api}/interaction`, data)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            dispatch(submitLFSuccess(response.data));
            dispatch(getStatistics());
            return response.data;
        })
        .catch(error => {
            dispatch(submitLFError(error));
        })
    }
}

export function getLFstats() {
    return dispatch => {
        dispatch(allLFPending({}));
        axios.get(`${api}/labelingfunctions`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            dispatch(lfStats(response.data));
            dispatch(getStatistics());
            return response.data;
        })
        .catch(error => {
            dispatch(lfStatsError(error));
        })
    }
}

export function getLFexamples(lf_id) {
    if (lf_id === null) {
        return dispatch => { dispatch({
                    type: LF_LABEL_EXAMPLES, 
                    data: {}
                })};
    }
    return dispatch => {
        dispatch(oneLFPending(lf_id));
        axios.get(`${api}/labelingfunctions/${lf_id}`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            dispatch({
                    type: LF_LABEL_EXAMPLES, 
                    data: response.data
                });
            return response.data;
        })
        .catch(error => {
            dispatch(lfStatsError(error));
        })    
    }
}

export default submitLFs;
import axios from 'axios';
import api from './api'

export const SAVE_ERROR = "SAVE_ERROR"
export const SAVE_PENDING = "SAVE_PENDING"
export const SAVE_SUCCESS = "SAVE_SUCCESS"


function saveSuccess(data) {
    return {
        type: SAVE_SUCCESS,
        data: data
    }
}

export function savePending() {
    return {
        type: SAVE_PENDING
    }
}

function saveError(error) {
    return {
        type: SAVE_ERROR,
        error: error
    }
}

export function saveModel() { 
    return dispatch => {
        window.open(`${api}/save`);
    }
}
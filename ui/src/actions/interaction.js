import axios from 'axios';
import api from './api'
import {setInteraction, getTextPending} from './getText'

export const DELETE_INTERACTION = "DELETE_INTERACTION";
export const GET_INTERACTION_SUCCESS = "GET_INTERACTION_SUCCESS"
export const GET_INTERACTION_PENDING = "GET_INTERACTION_PENDING"
export const GET_INTERACTION_ERROR = "GET_INTERACTION_ERROR"

export function deleteInteraction(index) {
    return {
        type: DELETE_INTERACTION,
        index: index
    }
}

function getInteractionError() {
    return {
        type: GET_INTERACTION_ERROR
    }
}

function getInteractionPending() {
    return {
        type: GET_INTERACTION_PENDING
    }
}

function getInteractionSuccess(data) {
    return {
        type: GET_INTERACTION_SUCCESS,
        data: data
    }
}

export function getInteraction(index) {
    return dispatch => {
        dispatch(getInteractionPending());
        axios.get(`${api}/interaction/${index}`)
        .then(response => {
            if(response.error) {
                dispatch(getInteractionError());
                throw(response.error);
            }
            dispatch(getInteractionSuccess(response.data));
            return response.data;
        })
        .catch(error => {
            dispatch(getInteractionError(error));
        })
    }
}

export function setAsCurrentInteraction(index) {
    return dispatch => {
        dispatch(getTextPending())
        axios.get(`${api}/interaction/${index}`)
        .then(response => {
            if(response.error) {
                dispatch(getInteractionError());
                throw(response.error);
            }
            setInteraction(response, dispatch);
            dispatch(getInteractionSuccess(response.data));
            return response.data;
        })
    }
}
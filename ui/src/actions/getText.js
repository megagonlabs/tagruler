import axios from 'axios';
import api from './api'
import { annotate, select_link, NER } from './annotate'
import { reset_label, label } from "./labelAndSuggestLF";

export const GET_TEXT_PENDING = "GET_TEXT_PENDING";
export const GET_TEXT_SUCCESS = 'GET_TEXT_SUCCESS';
export const GET_TEXT_ERROR = "GET_TEXT_ERROR";

export function getTextPending() {
    return {
        type: GET_TEXT_PENDING,
    }
}

function newText(data, index){
   return {
        type: GET_TEXT_SUCCESS, 
        data,
        index
    }
}

function getTextError(error) {
    return {
        type: GET_TEXT_ERROR,
        error
    }
}

export function getText(){
    return dispatch => {
        dispatch(getTextPending());
        axios.get(`${api}/interaction`)
        .then(response => {
            if (response.error) {
                throw(response.error);
            }
            setInteraction(response, dispatch);
            return response.data.text
        })
        .catch(error => {
            dispatch(getTextError(error));
        })
    }
}

// Set the new text, and reset all state related to the previous text
export function setInteraction(response, dispatch){
    //change the text
    dispatch(newText(response.data.text, response.data.index));

    //reset annotations
    let annotations = [];
    if ("annotations" in response.data) {
        annotations = response.data.annotations;
    }
    dispatch(annotate(annotations));
    let ners = [];
    if ("NER" in response.data) {
        ners = response.data.NER;
    }
    dispatch(NER(ners));

    //reset selected span to link
    dispatch(select_link({type: null}));

    if ("label" in response.data) {
        dispatch(label(response.data));
    } else {
        //reset selected label
        dispatch(reset_label());
    }

}
import axios from 'axios';
import api from './api'
import conceptStyler from './ConceptStyler'

export const GET_CONCEPTS_PENDING = "GET_CONCEPTS_PENDING";
export const UPDATE_CONCEPT_PENDING = "UPDATE_CONCEPT_PENDING";
export const GET_CONCEPTS_SUCCESS = "GET_CONCEPTS_SUCCESS";
export const GET_CONCEPTS_ERROR = "GET_CONCEPTS_ERROR";
export const SELECT_CONCEPT = 'SELECT_CONCEPT'

function getConceptsPending() {
    return {
        type: GET_CONCEPTS_PENDING,
    }
}

function updateConceptPending(conceptName) {
    return {
        type: UPDATE_CONCEPT_PENDING,
        conceptName: conceptName
    }
}

function getConceptsSuccess(data) {
    return {
        type: GET_CONCEPTS_SUCCESS,
        data: data
    }
}

function getConceptsError(error) {
    return {
        type: GET_CONCEPTS_ERROR,
        error: error
    }
}

function fetchConcepts() {
    return dispatch => {
        dispatch(getConceptsPending());
        axios.get(`${api}/concept`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            const conceptNames = Object.keys(response.data);
            let data = {};
            for (let i=0; i<conceptNames.length; i++) {
                let name = conceptNames[i];
                data[name] = {
                    "tokens": response.data[name], 
                    "color": conceptStyler.color(name),
                    "hotkey": conceptStyler.hotkey(name),
                    "pending": false
                };
            }
            dispatch(getConceptsSuccess(data))
            return data;
        })
        .catch(error => {
            dispatch(getConceptsError(error));
        })
    }
}

function addConcept(conceptName){
    const data = {
        name: conceptName,
        tokens: []
    };

    return dispatch => {
        dispatch(getConceptsPending());
        axios.post(`${api}/concept`, data)
        .then(response => { 
            if(response.error) {
                throw(response.error);
            }
            dispatch(fetchConcepts())
        });
    }
}

function deleteConcept(conceptName){
    return dispatch => {
        dispatch(updateConceptPending(conceptName));
        axios.delete(`${api}/concept/${conceptName}`)
        .then(response => { 
            if(response.error) {
                throw(response.error);
            }
            dispatch(fetchConcepts())
        });
    }
}

// add string to concept tokens list
function updateConcept(conceptName, data){
    return dispatch => {
        dispatch(updateConceptPending(conceptName));
        axios.put(`${api}/concept/${conceptName}`, data)
        .then(response => { 
            if(response.error) {
                throw(response.error);
            }
            dispatch(fetchConcepts())
        });
    }
}

// select a concept for annotating spans
export function select_concept(data){
    return dispatch => dispatch({type: SELECT_CONCEPT, data: data})
}

export const conceptEditors = {
    addConcept,
    deleteConcept,
    fetchConcepts,
    updateConcept
}
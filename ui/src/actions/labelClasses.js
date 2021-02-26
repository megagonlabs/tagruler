import axios from 'axios';
import api from './api'

export const GET_CLASSES_SUCCESS="GET_CLASSES_SUCCESS";
export const GET_CLASSES_PENDING="GET_CLASSES_PENDING";
export const GET_CLASSES_ERROR="GET_CLASSES_ERROR";
export const ADD_CLASS_SUCCESS="ADD_CLASS_SUCCESS";

const colorTheme = [
    {
        backgroundColor: 'rgb(255,236,153)',
        textColor: "rgb(239,140,2)",
    },
    {
        backgroundColor: "rgb(183,217,179)",
        textColor: "rgb(43,138,62)",
    },
    {
        backgroundColor: "rgb(201,231,239)",
        textColor: "rgb(12,152,173)",
    },
    {
        backgroundColor: "rgb(227,217,237)",
        textColor: "rgb(83,75,154)",
    },
    {
        backgroundColor: "rgb(248,200,201)",
        textColor: "rgb(225,49,49)",
    },
]

function pending() {
    return {
        type: GET_CLASSES_PENDING
    }
}

function getClassesSuccess(data) {
    return {
        type: GET_CLASSES_SUCCESS,
        data: data
    }
}

function addClassSuccess(data) {
    return {
        type: ADD_CLASS_SUCCESS,
        data: data
    }
}

function raiseError(error) {
    return {
        type: GET_CLASSES_ERROR,
        error: error
    }
}

function dataFromResponse(response) {
    return Object.keys(response.data).map(k => {
        return {
            key: parseInt(k), 
            name: response.data[k]
        }
    })
}


export function submitLabelsAndName(labelClasses, project_name=null) {
    return dispatch => {
        dispatch(pending());
        axios.post(`${api}/label`,
            {
                labels: labelClasses,
                name: project_name
            }
        )
        .then(response => {
            if (response.error) {
                throw(response.error);
            }
            const data = dataFromResponse(response.data);
            dispatch(getClassesSuccess(data));
        })
        .catch(error => {
            dispatch(raiseError(error));
        })
    }
}

export function addLabelClass(labelClassObj) {
    return dispatch => {
        dispatch(addClassSuccess(labelClassObj));   
    }
}

function fetchClasses() {
    return dispatch => {
        dispatch(pending());
        axios.get(`${api}/label`)
        .then(response => {
            if(response.error) {
                throw(response.error);
            }
            const data = dataFromResponse(response);
            for(let i=0;i<data.length;i++){
                data[i].backgroundColor = colorTheme[i].backgroundColor;
                data[i].textColor = colorTheme[i].textColor;
            }
            dispatch(getClassesSuccess(data));
        })
        .catch(error => {
            dispatch(raiseError(error));
        })
    }
}

export default fetchClasses;
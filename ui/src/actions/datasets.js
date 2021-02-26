import axios from 'axios';
import api from './api'

export const SUCCESS = "UPLOAD_SUCCESS"
export const PENDING = "UPLOAD_PENDING"
export const ERROR = "UPLOAD_ERROR"
export const SELECT = "SELECT_DATASET"


function UploadError(error) {
    return {
        type: ERROR,
        error: error
    }
}

function selectDataset(data) {
    return {
        type:SELECT,
        data: data
    }
}

function UploadPending() {
    return {
        type: PENDING
    }
}

function UploadSuccess(data) {
    return {
        type: SUCCESS,
        data: data
    }
}

export function fetchDatasets(idToken) {
    return dispatch => {
        axios.get(`${api}/datasets`, 
            { 
                headers: {
                    'Authorization': 'Bearer ' + idToken
                }
            }
        )
        .then(response => {
            if(response.error) {
                dispatch(UploadError());
                throw(response.error);
            }
            dispatch(UploadSuccess(response.data));
            return response.data;
        })
        .catch(error => {
            dispatch(UploadError(error));
        })
    }

}

export function setSelected(dataset_uuid, idToken=0) {
    return dispatch => {
        axios.post(`${api}/datasets`, 
            {dataset_uuid: dataset_uuid[0]},
            { 
                headers: {
                'Authorization': 'Bearer ' + idToken
                }
            }
        )
        .then(response => {
            if (response.error) {
                throw(response.error);
            }
            dispatch(selectDataset(dataset_uuid));
        })
    }
}

export default function uploadDataset(formData, dataset_uuid=0, idToken=0) {
    return dispatch => {
        dispatch(UploadPending());
        console.log(formData);
        axios.post(`${api}/datasets/${dataset_uuid}`, 
            formData, 
            { 
                headers: {
                'Content-Type': 'multipart/form-data',
                'Authorization': 'Bearer ' + idToken
                }
            }
        )
        .then(response => {
            if(response.error) {
                dispatch(UploadError());
                throw(response.error);
            }
            dispatch(UploadSuccess(response.data));
            setSelected(dataset_uuid);
            return response.data;
        })
        .catch(error => {
            dispatch(UploadError(error));
        })
    }
}
import axios from 'axios';
import api from './api'

function updateLoadingBar(value) {
    return {
        type: "LOADING_BAR",
        data: value
    }
}

function setThread(thread) {
    return {
        type: "SET_LAUNCH_THREAD",
        data: thread
    }
}

export function launchStatus(thread=0){
    return dispatch => {
        console.log("getting launch status");
        axios.get(`${api}/launch`)
        .then(response => {
            console.log(response);
            dispatch(updateLoadingBar(response.data));
        })
    }
}

export default function launch(){
    return dispatch => {
        dispatch(updateLoadingBar(0));
        dispatch(setThread(0));
        axios.post(`${api}/launch`, {})
        .then(response => {
            dispatch(setThread(response.data));
        });
    }
}

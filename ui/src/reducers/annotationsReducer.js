import { ANNOTATE, HIGHLIGHT, HIGHLIGHT_ERROR, SELECT_LINK, ADD_NER } from '../actions/annotate';
import { SPANANNOTATE } from '../actions/spanAnnotate';

export function spanAnnotations(state=[], action){
    switch (action.type){
        case SPANANNOTATE:
            return action.data
        default:
            return state;
    }
}

function annotations(state=[], action ){
    switch (action.type) {
        case ANNOTATE:
            return action.data
        default:
            return state
    }
}

export default annotations;

export function highlights(state={data: [], error: {}}, action ){
    switch (action.type) {
        case HIGHLIGHT:
            return {
                ...state, 
                data: action.data, 
                error: {},
                idx: null
            }
        case HIGHLIGHT_ERROR:
            var newErr = state.error;
            newErr[action.idx] = action.error;
            return {
                ...state,
                error: newErr
            }
        default:
            return state
    }
}

export function ners(state=[], action) {
    switch (action.type) {
        case ADD_NER:
            return action.data
        default:
            return state
    }
}

export function selectedLink(state={type: null}, action) {
    switch (action.type) {
        case SELECT_LINK: 
            return action.data
        default: 
            return state
    }
 }
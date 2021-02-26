import { LF_LABEL_EXAMPLES } from '../actions/submitLFs'

const initialState = {
}

export default function labelExamples(state=initialState, action) {
    switch(action.type) {
        case LF_LABEL_EXAMPLES:
            return action.data
        default:
            return state;
    }
}

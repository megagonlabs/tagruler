import { 
    NEW_LF,
    LABEL,
    SPANLABEL
} from '../actions/labelAndSuggestLF';
import { annotate, select_link, NER } from '../actions/annotate'

export function label (state = null, action ) {
    switch (action.type) {
        case LABEL:
            return action.data.label
        default:
            return state
    }
}

export function suggestedLF (state = {}, action) {
    switch (action.type) {
        case NEW_LF:
            /* Keep already selected LFs */
            const LF_names = Object.keys(state);
            var already_selected_lfs = {};
            for (var i = LF_names.length -1; i >= 0; i--) { 
                let lf_id = LF_names[i];               
                let lf = state[lf_id];
                if (lf.selected) {
                    already_selected_lfs[lf_id] = lf
                }
            }
            return {
                ...action.data,
                ...already_selected_lfs
                };
        default:
            return state;
    }
}
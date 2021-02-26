export const SPANANNOTATE = "SPANANNOTATE";

export function spanAnnotate(data) {
    return dispatch => {
        dispatch({
            type: SPANANNOTATE, 
            data
        })
    }
}
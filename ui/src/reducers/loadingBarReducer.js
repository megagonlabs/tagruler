export default function loadingBarReducer(state={progress: 0, thread: null}, action) {
    switch (action.type) {
        case "LOADING_BAR":
            if (action.data < 1.0) {
                return {
                    ...state,
                    progress: action.data,
                }
            } else {
                return {
                    ...state,
                    progress: action.data,
                    thread: null
                }
            }
        case "SET_LAUNCH_THREAD":
            return {
                ...state,
                thread: action.data
            }
        default:
            return state
    }
}
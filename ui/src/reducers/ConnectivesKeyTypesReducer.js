import { CONNECTIVE, KEYTYPE } from '../actions/connectivesAndKeyTypes'

const initialState = {
    fetched_conn: false,
    fetched_keytype: false,
    connective: {},
    keyType: {},
}

export default function connectivesKeyTypesReducer(state=initialState, action) {
    switch (action.type) {
        case CONNECTIVE:
            return {...state, connective: action.data, fetched_conn: true}
        case KEYTYPE:
            return {...state, keyType: action.data, fetched_keytype: true}
        default:
            return state
    }
}
export const ANNOTATE = "ANNOTATE";
export const HIGHLIGHT = "HIGHLIGHT";
export const HIGHLIGHT_ERROR = "HIGHLIGHT_ERROR"
export const SELECT_LINK = "SELECT_LINK";
export const ADD_NER = "ADD_NER";

export function annotate(data) {
    /* data is an array of annotations, for example
     [{
      id: 11,
      label: 0,
      start_offset: 11,
      end_offset: 17,
      text: 'Russia',
      link: null
    }]
  ] */
    return dispatch => {
        dispatch({
            type: ANNOTATE, 
            data
        })
    }
}

export function highlight(data) {
    /* data is an array of annotations, for example
     [{
      id: 11,
      label: 0,
      start_offset: 11,
      end_offset: 17,
      text: 'Russia',
      link: null
    }]
  ] */
    return dispatch => {
        dispatch({
            type: HIGHLIGHT, 
            data
        })
    }
}

export function NER(data) {
    return dispatch => {
        dispatch({
            type: ADD_NER, 
            data
        })
    }
}

export function highlight_regex(regex_str, text, case_sensitive, idx) {
    var data = [];
    var matches = [];
    try {
        var regex = null;
        if (!case_sensitive) {
          regex = new RegExp(regex_str, 'ig');
        } else {
          regex = new RegExp(regex_str, 'g');
        }
        matches = text.matchAll(regex);
    } 
    catch (err) {
        return dispatch => dispatch({
          type: HIGHLIGHT_ERROR, error: err.toString(), idx: idx});
    }
    for (const match of matches) {
        data.push({
            id: `${match.index}_pending`, // 'pending' because this annotation will not necessarily be submitted
            label: 'pending',
            start_offset: match.index,
            end_offset: match.index + match[0].length,
            link: null,
        });
    }
    return highlight(data);
}

export function isToken(string) {
  if (string.startsWith("(?:(?<=\\W)|(?<=^))(")) {
    string = string.slice("(?:(?<=\\W)|(?<=^))(".length);
    if (string.endsWith(")(?=\\W|$)")) {
      string = string.slice(0, string.length - ")(?=\\W|$)".length);
      return string;
    }
  } return false;
}

export function TokenToRegex(token) {
    const regex = "(?:(?<=\\W)|(?<=^))(" + token + ")(?=\\W|$)";
    return regex;
}

export function highlight_string(string, text, case_sensitive=false) {
    if (!case_sensitive) {
      text = text.toLowerCase();
    }
    const regex = TokenToRegex(string, case_sensitive);
  
    return highlight_regex(regex, text)
}

export function select_link(data){
    /* EXAMPLE DATA 
    {
    type: 'Undirected Link',
    segment: {
      id: 11,
      label: 0,
      start_offset: 11,
      end_offset: 15,
      text: 'Ever',
      link: null
    }
  }
  */
    return dispatch => {
        dispatch({
            type: SELECT_LINK, 
            data
        })
    }
}
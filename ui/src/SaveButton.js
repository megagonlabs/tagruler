import IconButton from '@material-ui/core/IconButton';
import SaveAltIcon from '@material-ui/icons/SaveAlt';

import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";
import {saveModel} from './actions/save'

const SaveButton = (props) => {
    return (
        <IconButton aria-label="save" onClick={props.save} color="inherit">
            <SaveAltIcon></SaveAltIcon>
        </IconButton>
    )
}


function mapStateToProps(state, ownProps?) {
    return {
    };
}

function mapDispatchToProps(dispatch) {
    return { 
        save: bindActionCreators(saveModel, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(SaveButton);

import React from 'react';
import {bindActionCreators} from 'redux';
import {connect} from "react-redux";

// actions
import getStatistics from './actions/getStatistics'
import submitLFs, { getLFstats } from './actions/submitLFs'
import { getText } from './actions/getText'
import {clear_suggestions} from "./actions/labelAndSuggestLF";
import { setAsCurrentInteraction } from './actions/interaction'

// presentational component
import NavigationButtons from './NavigationButtons'

class Navigation extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            queued_interactions: []
        };
    }

    selected_LFs(del=false) {
        const LF_ids = Object.keys(this.props.suggestedLF);
        var suggestedLF = this.props.suggestedLF;
        const selected_LFs = LF_ids.reduce( function (selected_LFs, id) {
            let LF = suggestedLF[id];
            if (LF.selected) {
                if (del) {
                    delete LF.selected;
                }
                selected_LFs[id] = LF;
            }
            return selected_LFs;
        }, {});
        return selected_LFs;
    }

    clickNext() {
        const selected_LFs = this.selected_LFs(true);
        const LFS_WILL_UPDATE = (Object.keys(selected_LFs).length > 0);
        if (LFS_WILL_UPDATE) {
            for (var i = selected_LFs.length - 1; i >= 0; i--) {
                let lf = selected_LFs[i];
                delete lf.selected;
            }
            this.props.submitLFs(selected_LFs);
        } 

        if (this.state.queued_interactions.length===0) {
            this.props.fetchNextText(); 
        } else {
            this.props.setAsCurrentInteraction(this.state.queued_interactions.pop());
        }
        this.props.clear_suggestions();
        this.props.getStatistics();
    }

    clickPrevious() {
        this.state.queued_interactions.push(this.props.index);
        this.props.setAsCurrentInteraction(this.props.index - 1);
    }

    render() {
        const selected_LFs = this.selected_LFs();
        const LFS_WILL_UPDATE = (Object.keys(selected_LFs).length > 0);
        const disableNext = LFS_WILL_UPDATE && this.props.LFLoading;
        const FabText = ((this.props.currentLabel === null) && (this.props.text.length !== 0)) ? "Skip" : "Next";
        
        return(<NavigationButtons 
            forward_text = {FabText} 
            clickNext={ disableNext ? null : this.clickNext.bind(this)} 
            clickPrevious={this.props.index > 0 ? this.clickPrevious.bind(this) : null}
        />)
    }
}

function mapStateToProps(state, ownProps?) {
    return {
        LFLoading: state.selectedLF.pending,
        text: state.text.data,
        suggestedLF: state.suggestedLF,
        currentLabel: state.label,
        index: state.text.index
    };
}
function mapDispatchToProps(dispatch) {
    return { 
        fetchNextText: bindActionCreators(getText, dispatch), 
        getStatistics: bindActionCreators(getStatistics, dispatch),
        submitLFs: bindActionCreators(submitLFs, dispatch),
        clear_suggestions: bindActionCreators(clear_suggestions, dispatch),
        getLFstats: bindActionCreators(getLFstats, dispatch),
        setAsCurrentInteraction: bindActionCreators(setAsCurrentInteraction, dispatch)
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(Navigation);

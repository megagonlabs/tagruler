import React from 'react';
import PropTypes from 'prop-types';
import {connect} from "react-redux";

import CancelIcon from '@material-ui/icons/Cancel';
import AddBoxIcon from '@material-ui/icons/AddBox';
import IndeterminateBoxIcon from '@material-ui/icons/IndeterminateCheckBox';
import IconButton from '@material-ui/core/IconButton';
import LinkIcon from '@material-ui/icons/Link';
import LinkOffIcon from '@material-ui/icons/LinkOff';
import Typography from '@material-ui/core/Typography';

import { InlineBox } from './RichTextUtils'
import { DIR_LINK, UNDIR_LINK } from './AnnotationBuilder'


import { styled } from '@material-ui/core/styles';
import Badge from '@material-ui/core/Badge';

import { colors } from '@material-ui/core';
import { MuiThemeProvider} from '@material-ui/core/styles';
import { createMuiTheme } from '@material-ui/core';
import { shadows } from '@material-ui/system';

const DeleteBadge = styled(Badge)({
    width: "fit-content",
    display: "inline",
    left: 10
});

const PositiveBadge = styled(Badge)({
    width: "fit-content",
    display: "inline",
    top: -10,
    left: 9,
});

const NegativeBadge = styled(Badge)({
    width: "fit-content",
    display: "inline",
    top: -10,
    left: 30
});

const LinkBadge = styled(Badge)({
    width: "fit-content",
    display: "inline",
    //top: 5
});

class SelectedSpan extends React.Component {
    constructor(props){
        super(props);

        this.handleClick = this.handleClick.bind(this);
        this.handleMouseHover = this.handleMouseHover.bind(this);
        this.handlePositiveMouseHover = this.handlePositiveMouseHover.bind(this);
        this.handleNegativeMouseHover = this.handleNegativeMouseHover.bind(this);
        
        this.state = {
            isHovering: false,
            isPosHovering: false,
            isNegHovering: false,
            isSpanPositive: true,
        };
    }

    handleBadgeClick(newState) {
        for (let i = 0; i < this.props.annotations.length; i++){
            if (this.props.annotations[i].start_offset == this.props.sid){
                this.props.annotations[i].isPositive = newState;
            }
        }
        this.setState({isSpanPositive: newState});
    }

    handleMouseHover(newState) {
        this.setState({isHovering: newState});
    }

    handlePositiveMouseHover(newState) {
        this.setState({isHovering: newState});
        this.setState({isPosHovering: newState});
    }

    handlePositiveMouseLeave(newState) {
        this.setState({isPosHovering: newState});
    }

    handleNegativeMouseHover(newState) {
        this.setState({isHovering: newState});
        this.setState({isNegHovering: newState});
    }

    handleNegativeMouseLeave(newState) {
        this.setState({isNegHovering: newState});
    }

    delayMouseLeave() {
        setTimeout(function() {
            this.setState({isHovering: false});
        }.bind(this), 1000);
    }

    handleClick(){
        this.props.annotate();
    }

    render(){
        const classes = this.props.classes;
        const linkVisible = ((this.state.isHovering) || (this.props.selectedLink.type===UNDIR_LINK));
        let style = this.props.style;

        const text = this.props.text;

        const innerSpan = (              
                    <InlineBox style={style} border={1} boxShadow={0} borderColor={this.state.isSpanPositive? "#228be6":"#f03e3e"} className={classes.box}
                        onMouseEnter={() => this.handleMouseHover(true)}
                        onMouseLeave={() => this.handleMouseHover(false)}
                        onClick={ ("clickSegment" in this.props) ? this.props.clickSegment : ()=>{} }
                    >
                    <Typography 
                        ref={this.textspan}
                        component="div"
                        className={ classes.text } 
                        id={this.props.id}
                        display="inline">
                    {text}
                    </Typography>
                    </InlineBox>
        );

        if ((this.props.clickLinkButton) && (this.props.onDelete)) {
            const themePosHover = createMuiTheme({
                palette:{
                    primary:{
                        main: "#228be6",
                    },
                    secondary:{
                        main: "#d0ebff",
                    }
                }
            })
            const themeNegHover = createMuiTheme({
                palette:{
                    primary:{
                        main: "#f03e3e"
                    },
                    secondary:{
                        main: "#ffc9c9"
                    }
                }
            })
            return (              
                    <>
                    <MuiThemeProvider theme={themePosHover}>
                        <PositiveBadge 
                        invisible={!this.state.isHovering} 
                        badgeContent={
                            <IconButton 
                                size="small"
                                onMouseEnter={() => this.handlePositiveMouseHover(true)}
                                onMouseLeave={() => this.handlePositiveMouseLeave(false)}
                                color={this.state.isPosHovering? "primary":(this.state.isSpanPositive? "primary":"secondary")}
                                onClick={() => this.handleBadgeClick(true)}><AddBoxIcon/></IconButton>
                        }>{""}</PositiveBadge>
                    </MuiThemeProvider>
                    <MuiThemeProvider theme={themeNegHover}>
                        <NegativeBadge 
                        invisible={!this.state.isHovering} 
                        badgeContent={
                            <IconButton 
                                size="small"
                                onMouseEnter={() => this.handleNegativeMouseHover(true)}
                                onMouseLeave={() => this.handleNegativeMouseLeave(false)}
                                color={this.state.isNegHovering? "primary":(!this.state.isSpanPositive? "primary":"secondary")}
                                //onMouseLeave={() => this.handleMouseHover(false)} 
                                onClick={() => this.handleBadgeClick(false)}><IndeterminateBoxIcon/></IconButton>
                        }>{""}</NegativeBadge>
                    </MuiThemeProvider>
                    {innerSpan}
                    <DeleteBadge 
                    invisible={!this.state.isHovering} 
                    badgeContent={
                        <IconButton 
                            size="small"
                            onMouseEnter={() => this.handleMouseHover(true)}
                            //onMouseLeave={() => this.handleMouseHover(false)} 
                            onClick={this.props.onDelete}><CancelIcon/></IconButton>
                    }>{""}</DeleteBadge>
                    </>);
        } else {
            return(innerSpan);
        }
    }
}

SelectedSpan.propTypes = {
    annotate: PropTypes.func,
    clickLinkButton: PropTypes.func,
    onDelete: PropTypes.func,
    classes: PropTypes.object,
    clickSegment: PropTypes.func

}

function mapStateToProps(state, ownProps?) {
    return { selectedLink: state.selectedLink,
        annotations: state.annotations };
}
function mapDispatchToProps(dispatch) {
    return {};
}

export default connect(mapStateToProps, mapDispatchToProps)(SelectedSpan);
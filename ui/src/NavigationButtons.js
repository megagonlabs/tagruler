import React from 'react';

// material UI
import ArrowIcon from '@material-ui/icons/ArrowForward';
import ArrowBackIcon from '@material-ui/icons/ArrowBack';
import ButtonGroup from '@material-ui/core/ButtonGroup';
import Button from '@material-ui/core/Button';

export default class NavigationButtons extends React.Component {
    render() {
        return(
            <ButtonGroup>
                <Button 
                    size="small" 
                    variant="contained" 
                    onClick={this.props.clickPrevious} 
                    disabled={this.props.clickPrevious===null}
                >
                    <ArrowBackIcon />
                    Previous
                </Button>
                <Button 
                    color="secondary" 
                    size="small" 
                    variant="contained" 
                    onClick={this.props.clickNext}
                    disabled={this.props.clickNext===null}
                >
                    { this.props.forward_text }
                    <ArrowIcon />
                </Button>
            </ButtonGroup>)
    }
}
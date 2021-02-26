import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import NavigationBar from './NavigationBar';
import LeftDrawer from './LeftDrawer';
import ProjectGrid from './ProjectGrid';
import Footer from './Footer';
import ProjectCreation from './ProjectCreation'

import {
  BrowserRouter as Router,
  Switch,
  Route,
} from "react-router-dom";

const styles = {
    grow: {
        flexGrow: 1,
    }
};

class Main extends React.Component {

    constructor(props) {
        super(props);

        this.handleDrawerClose = this.handleDrawerClose.bind(this);
        this.handleDrawerOpen = this.handleDrawerOpen.bind(this);

        this.drawerWidth = 200;
        this.state = {isDrawerOpen:false};
    }

    handleDrawerClose(){
        this.setState({isDrawerOpen:false} )
    }

    handleDrawerOpen(){
        this.setState({isDrawerOpen:true} )
    }

    render() {
        const classes = this.props.classes,
            isDrawerOpen = this.state.isDrawerOpen

        return (
            <Router>
                <div className={classes.grow} >
                    <CssBaseline/>
                    <NavigationBar handleDrawerOpen={this.handleDrawerOpen} isDrawerOpen={isDrawerOpen} />
                    <LeftDrawer handleDrawerClose={this.handleDrawerClose}  isDrawerOpen={isDrawerOpen} />
                        <Switch>
                          <Route path="/upload">
                            <ProjectCreation isDrawerOpen={isDrawerOpen} />
                          </Route>
                          <Route path="/project">
                            <ProjectGrid isDrawerOpen={isDrawerOpen} />
                          </Route>
                          <Route path="/">
                            <ProjectCreation isDrawerOpen={isDrawerOpen} />
                          </Route>
                        </Switch>
                        <Footer isDrawerOpen={isDrawerOpen} />
                </div>
            </Router>
        );
    }
}

export default withStyles(styles)(Main);

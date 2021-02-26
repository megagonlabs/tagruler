import React from 'react';
import Drawer from '@material-ui/core/Drawer';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import IconButton from '@material-ui/core/IconButton';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import { makeStyles, useTheme } from '@material-ui/core/styles';
import {default as TaskIcon} from '@material-ui/icons/Assignment' ;
import {default as DashboardIcon} from '@material-ui/icons/BarChart';
import {default as LabelsIcon} from '@material-ui/icons/Label';
import {default as DatasetIcon} from '@material-ui/icons/TextFormat';
import {default as SatisfiedIcon} from  '@material-ui/icons/SentimentSatisfied';
import {default as VerySatisfiedIcon} from  '@material-ui/icons/SentimentVerySatisfied';
import {default as MoodIcon} from  '@material-ui/icons/Mood';


const drawerWidth = 200;

const useStyles = makeStyles(theme => ({
    root: {
        display: 'flex',
    },
    menuButton: {
        marginRight: theme.spacing(2),
    },
    hide: {
        display: 'none',
    },
    drawer: {
        width: drawerWidth,
        flexShrink: 0,
    },
    drawerPaper: {
        width: drawerWidth,
    },
    drawerHeader: {
        display: 'flex',
        alignItems: 'center',
        padding: '0 8px',
        ...theme.mixins.toolbar,
        justifyContent: 'flex-end',
    },
    content: {
        flexGrow: 1,
        padding: theme.spacing(3),
        transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
        marginLeft: -drawerWidth,
    },
    contentShift: {
        transition: theme.transitions.create('margin', {
            easing: theme.transitions.easing.easeOut,
            duration: theme.transitions.duration.enteringScreen,
        }),
        marginLeft: 0,
    },
}));


const LeftDrawer = (props) => {

    const theme = useTheme(),
        classes = useStyles();



    return(
        <Drawer
            className={classes.drawer}
            variant="persistent"
            anchor="left"
            open={props.isDrawerOpen}
            classes={{
                paper: classes.drawerPaper,
            }}
        >
            <div className={classes.drawerHeader}>
                <IconButton onClick={props.handleDrawerClose}>
                    {theme.direction === 'ltr' ? <ChevronLeftIcon/> : <ChevronRightIcon/>}
                </IconButton>
            </div>
            <Divider/>
            <List>
                {['Dataset', 'Labels', 'Task Guidelines', 'Dashboard'].map((text, index) => (
                    <ListItem button key={text}>
                        <ListItemIcon>{index === 0 ? <DatasetIcon/> :
                            index === 1 ? <LabelsIcon/> :
                                index === 2 ? <TaskIcon/> :
                                    <DashboardIcon/>}
                        </ListItemIcon>
                        <ListItemText primary={text}/>
                    </ListItem>
                ))}
            </List>
            <Divider/>
            <List>
                {['Other', 'Labeling', 'Tasks'].map((text, index) => (
                    <ListItem button key={text}>
                        <ListItemIcon>{index === 0 ? <SatisfiedIcon/> :
                            index === 1 ? <VerySatisfiedIcon/> :
                                <MoodIcon/>}
                        </ListItemIcon>
                        <ListItemText primary={text}/>
                    </ListItem>
                ))}
            </List>

        </Drawer>
    );

};

export default  LeftDrawer;


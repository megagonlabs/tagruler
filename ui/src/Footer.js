import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import FavoriteIcon from '@material-ui/icons/Favorite';
import GithubIcon from './GithubIcon';
import Toolbar from '@material-ui/core/Toolbar';
import clsx from 'clsx';
import IconButton from "@material-ui/core/IconButton";

const drawerWidth = 200;

const useStyles = makeStyles(theme =>  ({
    root: {
        position:'fixed',
        bottom:0,
        left:0,
        width:'100%',
        justifyContent: 'center',
        alignItems: 'center',
        background:"rgba(50, 50, 50, 1)"
    },
    footer: {
        transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }),
    },
    footerShift: {
        width: `calc(100% - ${drawerWidth}px)`,
        marginLeft: drawerWidth,
        transition: theme.transitions.create(['margin', 'width'], {
            easing: theme.transitions.easing.easeOut,
            duration: theme.transitions.duration.enteringScreen,
        })
    },
    menuButton: {
        marginRight: theme.spacing(2),
    },
    icon:{
        color:'white'
    }

}));


function Footer(props) {

    const classes = useStyles(),
        isDrawerOpen = props.isDrawerOpen;

    function handleClick(url) {
        const win = window.open(url, '_blank');
        win.focus();
    }

    return (
        <Toolbar className={clsx(classes.root, classes.footer, { [classes.footerShift]: isDrawerOpen })} color={'primary'}>
            <IconButton edge="end" className={classes.menuButton} onClick={handleClick.bind(null,'https://github.com/')} >
                <GithubIcon/>
            </IconButton>
        </Toolbar>
    );
}

export default Footer;

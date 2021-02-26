import React from 'react';
import PropTypes from 'prop-types';

import CancelIcon from '@material-ui/icons/Cancel';
import Badge from '@material-ui/core/Badge';
import IconButton from '@material-ui/core/IconButton';

const defaultAnchor = { x: 0.5, y: 0 };
const defaultBorderColor = '#6A747E';
const defaultBorderWidth = 2;
const defaultOffset = 30;
const defaultOpacity = 0.25


export default class Link extends React.Component {
    constructor(props){
        super(props);

        this.state = {
            isHovering: true,
            width: 0,
            height: 0
        };

        this.updateDimensions = this.updateDimensions.bind(this);
    }

    componentDidMount() {
        window.addEventListener('resize', this.updateDimensions);
    }

    componentWillUnmount() {
        window.removeEventListener('resize', this.updateDimensions);
    }

    findElement(id) {
        return document.getElementById(id);
    }

    updateDimensions() {
        this.setState({ width: window.innerWidth, height: window.innerHeight });
    }

    detect() {
        const { from, to} = this.props;

        const a = this.findElement(from);
        const b = this.findElement(to);

        if (!a || !b) {
            console.error(`Elements with ids ${from} and ${to} were not found.`);
            return false;
        }

        const anchor0 = defaultAnchor;
        const anchor1 = defaultAnchor;

        const box0 = a.getBoundingClientRect();
        const box1 = b.getBoundingClientRect();

        let offsetX = window.pageXOffset;
        let offsetY = window.pageYOffset;

        let x0 = box0.left + box0.width * anchor0.x + offsetX;
        let x1 = box1.left + box1.width * anchor1.x + offsetX;
        const y0 = box0.top + box0.height * anchor0.y + offsetY;
        const y1 = box1.top + box1.height * anchor1.y + offsetY;

        if (Math.abs(x0-x1) < 30) {
            x0 = box0.left + offsetX;
            x1 = box1.left + box1.width + offsetX;
        }
        return { x0, y0, x1, y1 };
    }

    handleMouseHover(newState) {
        this.setState({isHovering: newState});
    }

    render() {
        this.props.classes.anchorOriginTopRightRectangle = {
            bottom: 0,
            right: 0,
            transform: 'scale(1) translate(0%, -50%)',
            transformOrigin: '100% 100%',
            '&$invisible': {
                transform: 'scale(0) translate(50%, 50%)',
            },
        }

        const classes = this.props.classes;
        let offset = this.props.offset || defaultOffset;

        const points = this.detect();
        let {x0, y0, x1, y1} = points;
        if (!points) {
            return(
                <svg/>
            )
        }
        
        const leftOffset = Math.min(x0, x1);
        const topOffset = Math.min(y0, y1) - offset;

        x0 -= leftOffset;
        x1 -= leftOffset;
        y0 -= topOffset;
        y1 -= topOffset;

        const width = Math.abs(x0 - x1);

        const positionStyle = {
            position: 'absolute',
            top: `${topOffset}px`,
            left: `${leftOffset}px`,
            width: width,
            height: Math.abs(y0 - y1) + offset
        }

        const color = this.props.color || defaultBorderColor;
        const strokeWidth = this.props.width || defaultBorderWidth;
        const strokeOpacity = this.props.opacity || defaultOpacity;

        return (
            <Badge 
                className={classes.link.badge}
                badgeContent={
                    <IconButton 
                        onMouseEnter={() => this.handleMouseHover(true)}
                        onMouseLeave={() => this.handleMouseHover(false)} 
                        onClick={this.props.onDelete}>
                    <CancelIcon/>
                    </IconButton>
                } 
                invisible={this.props.onDelete ? false : true} 
                style={positionStyle}>
                <svg style={{width: width}}>
                    <path 
                        d={` M ${x0} ${y0} \
                             Q ${(x0+x1)/2} -${offset-5} ${x1} ${y1}`}
                        stroke = {color}
                        strokeWidth={strokeWidth}
                        fill="none"
                        strokeOpacity={strokeOpacity}
                        onMouseEnter={() => this.handleMouseHover(true)}
                        onMouseLeave={() => this.handleMouseHover(false)}/>
                
                </svg>
            </Badge>)
    }
}

Link.propTypes = {
    from: PropTypes.string.isRequired,
    to: PropTypes.string.isRequired,
    width: PropTypes.number,
    color: PropTypes.string,
    onDelete: PropTypes.func,
    offSet: PropTypes.number,
    classes: PropTypes.object
}
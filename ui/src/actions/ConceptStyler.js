class ConceptStyler{

  constructor() {
    this.colorIndex = 0;
    this.colorAssignments = {};
    this.hotKeys = {}; 
    this.availableLetters = "abcdefghijklmnopqrstuvwxyz-=[];',.";
  }

  hotkey(conceptString) {
    if (conceptString in this.hotKeys) {
      return this.hotKeys[conceptString];
    } 

    let letter = conceptString[0];
    if (this.availableLetters.indexOf(letter) === -1) {
      letter = this.availableLetters[0];
    }
    this.availableLetters = this.availableLetters.replace(letter, "");
    this.hotKeys[conceptString] = letter;
    return letter;
  }

  color(conceptString) {
    return this.nextColor(conceptString);
  }

  nextColor(string){
    if (string in this.colorAssignments){
      return this.colorAssignments[string];
    }
    const colorPalette =  ["#2CA02C", "#E377C2", "#17BECF", "#8C564B",  "#D62728", "#BCBD22", "#9467BD", "#FF7F0E", "#1F77B4"];
    let color = colorPalette[this.colorIndex];
    this.colorAssignments[string] = color;
    this.colorIndex = (this.colorIndex + 1) % colorPalette.length;
    return color
  }

  stringToColor(string) {
    let hash = 0;
    let i;

    /* eslint-disable no-bitwise */
    for (i = 0; i < string.length; i += 1) {
      hash = string.charCodeAt(i) + ((hash << 5) - hash);
    }

    let colour = '#';

    for (i = 0; i < 3; i += 1) {
      const value = (hash >> (i * 8)) & 0xff;
      colour += `00${value.toString(16)}`.substr(-2);
    }
    /* eslint-enable no-bitwise */

    return colour;
  }
}

let conceptStyler = new ConceptStyler();
export default conceptStyler;
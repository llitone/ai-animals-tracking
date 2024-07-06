const express = require('express')
const path = require('path');
const port = 5001;
// const cookieParser = require('cookie-parser');
const logger = require('morgan');
const { create } = require('express-handlebars');

const main = require("./routes/main");

const app = express()
const hbs = create({
    extname: 'hbs',
    defaultLayout: 'layout',
    layoutsDir: __dirname + '/views/layouts/',
    partialsDir: __dirname + '/views/partials/'
});
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
app.engine('hbs', hbs.engine);
app.set('view engine', 'hbs');
app.set('views', './views');
app.use(express.static(__dirname + '/public'));
app.set('views', path.join(__dirname, 'views'));

app.use('/', main)

module.exports = app;
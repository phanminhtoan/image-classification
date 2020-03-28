const PORT = process.env.PORT || 3000;
const HOSTNAME = '0.0.0.0';

const express = require('express');
const path = require('path');
const cors = require('cors');

let app = express();

var corsOptions = {
  origin: '*'
}

// config default proxy
app.use(express.static(__dirname));

app.get("/", cors(corsOptions), (req, res, next) => {
	// res.header("Access-Control-Allow-Origin", "https://0.0.0.0:5000"); // update to match the domain you will make the request from
  	res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
	res.sendFile(path.join(__dirname+"/index.html"));
	next();
});

app.listen(PORT, (err) => {
	if (err) {
		console.log(err);
	} else {
		console.log("Website is Up on PORT %d and HOSTNAME %s !", PORT, app.get('env'));
	}
});
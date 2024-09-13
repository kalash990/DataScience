require("dotenv").config();

const PORT = process.env.PORT || 443;
const HOST = process.env.HOSTNAME || "testnodeweb-erf3exc0bnchhdb5.eastus-01.azurewebsites.net";

const path = require("path");
const hbs = require("hbs");
const express = require("express");
const app = express();

const webRouter = require("./routers/webrouter");
const apiRouter = require("./routers/apirouter");

app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, "static")));
app.set("view engine", "hbs");
app.set("views", path.join(__dirname, "views"));

hbs.registerPartials(path.join(__dirname, 'views', 'partials'));

app.use("/", webRouter);
app.use("/api", apiRouter);

app.listen(PORT, () => {
    console.log(`Server running at ${HOST}:${PORT}`);
});


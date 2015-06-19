var gulp = require("gulp");
var serve = require("gulp-serve");
var babel = require("gulp-babel");
var babelify = require("babelify");
var browserify = require("browserify");
var source = require("vinyl-source-stream");
var gutil = require("gulp-util");
var concat = require("gulp-concat");
var bower = require("gulp-bower")

gulp.task("babelify", function(){
	browserify({
		entries: "./app.es6",
		debug: true
	})
	.transform(babelify.configure({
	  blacklist: ["regenerator"],
	  compact: false
	}))
	.bundle()
	.on("error", gutil.log)
	.pipe(source("main.js"))
	.pipe(gulp.dest("./dist"));
});

gulp.task("watch", function(){
	gulp.watch(["app.es6", "src/**/*.es6"], ["babelify"]);
});

gulp.task('bower', function() {
  return bower({ cmd: 'update'});
});

gulp.task("serve", serve("."));

gulp.task("default", ["watch", "babelify", "serve"])
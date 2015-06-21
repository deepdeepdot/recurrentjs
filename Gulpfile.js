var gulp = require("gulp");
var serve = require("gulp-serve");
var babel = require("gulp-babel");
var babelify = require("babelify");
var browserify = require("browserify");
var source = require("vinyl-source-stream");
var gutil = require("gulp-util");
var concat = require("gulp-concat");
var bower = require("gulp-bower");
var sass = require('gulp-sass');

gulp.task("babelify", function(){
	browserify({
		entries: "./src/js/app.es6",
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


gulp.task('bower', function() {
  return bower({ cmd: 'update'});
});

gulp.task('sass', function () {
  gulp.src('./src/css/**/*.scss')
    .pipe(sass().on('error', sass.logError))
    .pipe(gulp.dest('./dist'));
});

gulp.task("watch", function(){
	gulp.watch(["src/js/**/*.es6"], ["babelify"]);
	gulp.watch(["src/css/**/*.scss"], ["sass"]);
});

gulp.task("serve", serve("."));

gulp.task("default", ["watch", "babelify", "sass", "serve"])
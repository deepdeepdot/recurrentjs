var fs = require("fs");
var gulp = require("gulp");
var serve = require("gulp-serve");
var babel = require("gulp-babel");
var babelify = require("babelify");
var browserify = require("browserify");
var source = require("vinyl-source-stream");
var gutil = require("gulp-util");
var concat = require("gulp-concat");
var sass = require('gulp-sass');
var template = require('gulp-template');
var rename = require('gulp-rename');

gulp.task("babelify", function(){
  browserify({
    entries: "./src/js/app.js",
    debug: true
  })
  .transform(babelify.configure({
    stage: 1,
    compact: false
  }))
  .bundle()
  .on("error", gutil.log)
  .pipe(source("app.js"))
  .pipe(gulp.dest("./dist"));
});

gulp.task('sass', function() {
  gulp.src('./src/css/**/*.scss')
    .pipe(sass().on('error', sass.logError))
    .pipe(gulp.dest('./dist'));
});

gulp.task('compile_index', function() {
  var input_files = fs.readdirSync(__dirname + "/data");

  gulp.src('./src/index.html.tmpl')
    .pipe(template({
      input_files: input_files
    }))
    .pipe(rename('index.html'))
    .pipe(gulp.dest('.'));
});

gulp.task("watch", function(){
  gulp.watch(["src/js/**/*.js"], ["babelify"]);
  gulp.watch(["src/css/**/*.scss"], ["sass"]);
  gulp.watch(["data/*", "src/index.html.tmpl"], ["compile_index"]);
});

gulp.task("serve", serve("."));

gulp.task("default", ["watch", "babelify", "sass", "compile_index", "serve"])
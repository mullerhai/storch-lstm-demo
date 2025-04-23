ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "3.6.4"

lazy val root = (project in file("."))
  .settings(
    name := "storch-lstm-demo"
  )
libraryDependencies += "io.github.mullerhai" % "core_3" % "0.2.3-1.15.1"
libraryDependencies += "io.github.mullerhai" % "vision_3" % "0.2.3-1.15.1"
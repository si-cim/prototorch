pipeline {
  agent none
  stages {
    stage('Tests') {
      agent {
        docker {
          image 'python:3.9'
        }

      }
      steps {
        sh 'pip install .[all] --progress-bar off'
      }
    }

  }
}
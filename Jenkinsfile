pipeline {
  agent none
  stages {
    stage('Tests') {
      agent {
        docker {
          image 'python:3.9'
          args '--user 0:0'
        }

      }
      steps {
        sh 'whoami'
        sh 'pip install .[all] --user --progress-bar off'
      }
    }

  }
}

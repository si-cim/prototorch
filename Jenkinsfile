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
        sh 'pip install pip --upgrade --progress-bar off'
        sh 'pip install .[all] --progress-bar off'
      }
    }

  }
}
